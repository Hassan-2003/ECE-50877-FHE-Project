"""
FIDESlib emitter: translates the OpenFHE C++ that HEIR generates into FIDESlib C++.

HEIR has no FIDESlib backend. FIDESlib (GPU CKKS) mirrors OpenFHE's API in namespace
`fideslib`, so the OpenFHE code HEIR emits (heir-translate --emit-openfhe-pke) ports with
a handful of mechanical rewrites, the same route the research plan names ("porting generated
OpenFHE code"):

  * includes / namespace:  openfhe.h + lbcrypto  ->  <fideslib.hpp> + fideslib
    (EvalKey / ConstCiphertext aliases, which FIDESlib lacks, are dropped or inlined)
  * cc->ModReduce*(...)    ->  cc->Rescale*(...)     (FIDESlib's name for the same op)
  * ring dimension query   ->  cc->GetRingDimension()  (no GetCryptoParameters())
  * decrypt inputs are non-const (FIDESlib Decrypt takes Ciphertext&)
  * GPU setup in the generated crypto context: SetDevices, plaintext/ciphertext autoload
  * a `main()` driver (keygen on CPU, LoadContext to the GPU, timed encrypt / eval /
    decrypt with cudaDeviceSynchronize) that speaks the same tiny text protocol as
    Scripts/fideslib_backend.py.

Every `cc->Method(` the kernel uses is checked against the methods declared in FIDESlib's
api/CryptoContext.hpp; anything missing raises UnsupportedPrimitive instead of emitting code
that will not build.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIDES_API = os.path.join(ROOT, "env", "src", "FIDESlib", "api", "CryptoContext.hpp")

# Used when the FIDESlib sources are not checked out (matches FIDESlib 2.2.0).
FALLBACK_METHODS = {
    "Enable", "GetRingDimension", "KeyGen", "EvalMultKeyGen", "EvalRotateKeyGen",
    "MakeCKKSPackedPlaintext", "Encrypt", "Decrypt", "EvalAdd", "EvalAddInPlace",
    "EvalAddMutable", "EvalSub", "EvalSubInPlace", "EvalMult", "EvalMultInPlace",
    "EvalMultMutable", "EvalRotate", "EvalRotateInPlace", "EvalFastRotationPrecompute",
    "EvalFastRotation", "EvalFastRotationExt", "Rescale", "RescaleInPlace", "LoadContext",
}

# OpenFHE name -> FIDESlib name
RENAMES = {"ModReduce": "Rescale", "ModReduceInPlace": "RescaleInPlace"}


class UnsupportedPrimitive(Exception):
    pass


def fideslib_methods():
    if not os.path.exists(FIDES_API):
        return FALLBACK_METHODS
    return set(re.findall(r"\b(\w+)\(", open(FIDES_API).read())) | FALLBACK_METHODS


DRIVER = r"""
// ---------------------------------------------------------------- driver (emitted)
// Required: without it the timings would miss the syncs and GPU memory would read 0.
#include <cuda_runtime.h>
#define FIDES_SYNC() cudaDeviceSynchronize()
static size_t gpu_used_bytes() {
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  return total_b - free_b;
}
#include <chrono>

static long peak_rss_bytes() {
  std::ifstream f("/proc/self/status");
  std::string l;
  while (std::getline(f, l))
    if (l.rfind("VmHWM:", 0) == 0) return std::stol(l.substr(6)) * 1024;
  return 0;
}

int main(int argc, char** argv) {
  using clk = std::chrono::steady_clock;
  auto secs = [](clk::time_point a) {
    return std::chrono::duration<double>(clk::now() - a).count();
  };
  if (argc < 2) { std::cerr << "usage: " << argv[0] << " input.txt [repeats]\n"; return 2; }
  int repeats = argc > 2 ? std::atoi(argv[2]) : 1;
  std::ifstream in(argv[1]);
  size_t n = 0;
  in >> n;
  std::vector<float> x(n);
  for (auto& v : x) in >> v;

  size_t gpu0 = gpu_used_bytes();
  auto t0 = clk::now();
  auto cc = KERNEL__generate_crypto_context();
  auto keys = cc->KeyGen();
  cc->EvalMultKeyGen(keys.secretKey);
  cc = KERNEL__configure_crypto_context(cc, keys.secretKey);
  cc->LoadContext(keys.publicKey);
  FIDES_SYNC();
  double t_setup = secs(t0);

  t0 = clk::now();
  auto ct = KERNEL__encrypt__arg0(cc, x, keys.publicKey);
  FIDES_SYNC();
  double t_enc = secs(t0);

  // One untimed warm-up run (first-call GPU / allocator costs), then `repeats` timed runs.
  // The kernel takes the input as const, so every run sees the same ciphertext.
  auto out = KERNEL(cc, ct);
  FIDES_SYNC();
  std::vector<double> t_eval;
  for (int r = 0; r < repeats; ++r) {
    t0 = clk::now();
    out = KERNEL(cc, ct);
    FIDES_SYNC();
    t_eval.push_back(secs(t0));
  }

  t0 = clk::now();
  auto y = KERNEL__decrypt__result0(cc, out, keys.secretKey);
  FIDES_SYNC();
  double t_dec = secs(t0);

  std::cout.precision(9);
  std::cout << "setup_s=" << t_setup << "\nencrypt_s=" << t_enc << "\neval_s=";
  for (double t : t_eval) std::cout << t << " ";
  std::cout << "\ndecrypt_s=" << t_dec << "\npeak_rss=" << peak_rss_bytes()
            << "\ngpu_bytes=" << (gpu_used_bytes() - gpu0) << "\ny=";
  for (float v : y) std::cout << v << " ";
  std::cout << "\n";
  return 0;
}
"""


def emit_fideslib(openfhe_cpp, func="kernel"):
    """OpenFHE C++ from HEIR (entry function `func`) -> complete FIDESlib program source."""
    src = openfhe_cpp.replace(
        "cc->GetCryptoParameters()->GetElementParams()->GetRingDimension()",
        "cc->GetRingDimension()")

    # Unsupported-primitive check, after renaming.
    called = set(re.findall(r"\bcc->(\w+)\(", src))
    mapped = {RENAMES.get(c, c) for c in called}
    missing = sorted(mapped - fideslib_methods())
    if missing:
        raise UnsupportedPrimitive(
            "FIDESlib has no equivalent for: " + ", ".join(missing))

    # Includes + namespace.
    src = re.sub(r'#include\s+"[^"]*openfhe[^"]*".*\n', "#include <fideslib.hpp>\n", src)
    src = src.replace("using namespace lbcrypto;", "using namespace fideslib;")
    src = re.sub(r"using EvalKeyT = .*\n", "", src)   # FIDESlib has no EvalKey type
    src = src.replace("ConstCiphertext<DCRTPoly>", "const Ciphertext<DCRTPoly>")
    src = '#include <algorithm>\n#include <complex>\n#include <string>\n' + src

    # Renamed / removed OpenFHE-only calls.
    for old, new in RENAMES.items():
        src = re.sub(rf"\bcc->{old}\(", f"cc->{new}(", src)

    # FIXEDMANUAL (below) encodes fresh plaintexts at 2^scaleModSize. A vector whose entries
    # are all below one unit at that scale (FIR edge taps of ~1e-18) makes OpenFHE's encoder
    # throw "Scaling factor too small"; such entries round to 0 anyway, so encode them as 0.
    src = src.replace("using namespace fideslib;", "using namespace fideslib;\n"
                      "static void fides_flush_tiny(std::vector<double>& v) {\n"
                      "  for (auto& x : v) if (std::abs(x) < 1e-15) x = 0.0;\n}", 1)
    src = re.sub(r"^([ \t]*)(.*= cc->MakeCKKSPackedPlaintext\((\w+)\);)",
                 r"\1fides_flush_tiny(\3);\n\1\2", src, flags=re.M)

    # Decrypt takes a non-const Ciphertext&: copy instead of binding const refs.
    def fix_decrypt(m):
        return m.group(0).replace("const auto& ", "auto ")
    src = re.sub(rf"std::vector<float> {func}__decrypt__result\d+\(.*?\n}}\n", fix_decrypt,
                 src, flags=re.S)

    # HEIR's OpenFHE loops (e.g. hoisted rotations) are OpenMP-parallel on the CPU; on the
    # GPU every iteration would share one device context and lazily load the same input
    # ciphertext, so run them sequentially (the GPU parallelizes inside each op).
    src = re.sub(r"[ \t]*#pragma omp parallel for.*\n", "", src)

    # GPU parameters on the generated crypto context. HEIR emits explicit ModReduce calls
    # and sets no scaling technique: OpenFHE's default FLEXIBLEAUTOEXT turns those into
    # no-ops, but FIDESlib's RescaleInPlace always rescales, so use the manual technique the
    # explicit schedule is written for.
    src = re.sub(r"(CCParamsT params;\n)",
                 r"\1  params.SetDevices({0});\n  params.SetPlaintextAutoload(true);\n"
                 r"  params.SetCiphertextAutoload(true);\n"
                 r"  params.SetScalingTechnique(FIXEDMANUAL);\n", src, count=1)

    return src + DRIVER.replace("KERNEL", func)
