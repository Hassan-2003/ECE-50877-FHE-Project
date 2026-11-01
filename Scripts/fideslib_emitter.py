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
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define FIDES_SYNC() cudaDeviceSynchronize()
static size_t gpu_used_bytes() {
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  return total_b - free_b;
}
#else
#define FIDES_SYNC() ((void)0)
static size_t gpu_used_bytes() { return 0; }
#endif
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
  if (argc < 2) { std::cerr << "usage: " << argv[0] << " input.txt\n"; return 2; }
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

  t0 = clk::now();
  auto out = KERNEL(cc, ct);
  FIDES_SYNC();
  double t_eval = secs(t0);

  t0 = clk::now();
  auto y = KERNEL__decrypt__result0(cc, out, keys.secretKey);
  FIDES_SYNC();
  double t_dec = secs(t0);

  std::cout.precision(9);
  std::cout << "setup_s=" << t_setup << "\nencrypt_s=" << t_enc << "\nfhe_time_s=" << t_eval
            << "\ndecrypt_s=" << t_dec << "\npeak_rss=" << peak_rss_bytes()
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

    # Decrypt takes a non-const Ciphertext&: copy instead of binding const refs.
    def fix_decrypt(m):
        return m.group(0).replace("const auto& ", "auto ")
    src = re.sub(rf"std::vector<float> {func}__decrypt__result\d+\(.*?\n}}\n", fix_decrypt,
                 src, flags=re.S)

    # GPU parameters on the generated crypto context.
    src = re.sub(r"(CCParamsT params;\n)",
                 r"\1  params.SetDevices({0});\n  params.SetPlaintextAutoload(true);\n"
                 r"  params.SetCiphertextAutoload(true);\n", src, count=1)

    return src + DRIVER.replace("KERNEL", func)
