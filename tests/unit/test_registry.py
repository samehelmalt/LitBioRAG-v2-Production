import pytest

from models.registry import ModelRegistry, ModelSpec, RegistryError
from models.registry.loader import DEFAULT_PATH, registered_adapters
from models.registry.schema import model_family

YAML = """
default_profile: a
profiles:
  a:
    require_calibrated: true
    models:
      dense_encoder: hash:64
      reranker: lexical:x
      generator:
        kind: hf_causal
        model_id: Qwen/Qwen3-8B
        quantization: 4bit
      verifier:
        kind: hf_seq_cls
        model_id: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
        calibration:
          for_model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
          thresholds: {entail: 0.62, contradict: 0.55}
  shared_family:
    models:
      generator: {kind: hf_causal, model_id: Qwen/Qwen3-8B}
      verifier: {kind: hf_seq_cls, model_id: Qwen/Qwen3-Reranker-0.6B}
"""


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(YAML)
    return p


def test_spec_parsing_and_family():
    s = ModelSpec.parse("openai_compatible:Qwen/Qwen3-14B@v1?endpoint=http://x:8000/v1&max_tokens=2048&temperature=0.1", "generator")  # noqa: E501
    assert s.model_id == "Qwen/Qwen3-14B" and s.revision == "v1" and s.endpoint == "http://x:8000/v1"
    assert s.max_tokens == 2048 and s.params == {"temperature": "0.1"} and s.family == "qwen"
    assert model_family("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli") == "deberta"
    assert model_family("ncbi/MedCPT-Cross-Encoder") == "medcpt"
    with pytest.raises(ValueError):
        ModelSpec.parse("hash:128", "generator")  # kind not valid for role
    with pytest.raises(ValueError):
        ModelSpec.parse("nonsense", "generator")


def test_profile_load_env_override_and_thresholds(cfg):
    reg = ModelRegistry.load(cfg, env={})
    assert reg.profile.name == "a" and reg.spec("dense_encoder").model_id == "64"
    assert reg.thresholds("verifier") == {"entail": 0.62, "contradict": 0.55}
    with pytest.raises(RegistryError):  # generator has no calibration and profile requires it
        reg.thresholds("generator")
    # swapping the verifier by env var invalidates its thresholds without touching code
    reg2 = ModelRegistry.load(
        cfg, env={"LITBIORAG_MODEL__VERIFIER": "hf_seq_cls:cross-encoder/nli-deberta-v3-large"}
    )
    assert reg2.spec("verifier").model_id == "cross-encoder/nli-deberta-v3-large"
    with pytest.raises(RegistryError):
        reg2.thresholds("verifier")
    # env override for a test double is accepted and needs no calibration
    reg3 = ModelRegistry.load(cfg, env={"LITBIORAG_MODEL__VERIFIER": "overlap:x"})
    assert reg3.thresholds("verifier") == {}


def test_family_independence_rule(cfg):
    with pytest.raises(RegistryError, match="share model family"):
        ModelRegistry.load(cfg, profile="shared_family", env={})
    with pytest.raises(RegistryError):
        ModelRegistry.load(cfg, profile="missing", env={})


def test_build_test_doubles_and_missing_adapter(cfg):
    reg = ModelRegistry.load(cfg, env={})
    enc = reg.build("dense_encoder")
    assert enc.name == "hash:64" and enc.dim == 64 and reg.build("dense_encoder") is enc
    assert reg.build("reranker").name == "lexical-overlap"
    assert ("hash", "dense_encoder") in registered_adapters()
    no_adapter = ModelRegistry.load(cfg, env={"LITBIORAG_MODEL__VERIFIER": "minicheck:x"})
    with pytest.raises(RegistryError, match="no adapter registered"):
        no_adapter.build("verifier")  # 'minicheck' is a valid kind with no adapter yet


def test_shipped_models_yaml_profiles_load():
    for name in ("test", "thesis-baseline", "default"):
        reg = ModelRegistry.load(DEFAULT_PATH, profile=name, env={})
        assert set(reg.roles()) >= {"dense_encoder", "reranker", "generator", "verifier"}
        d = reg.describe()
        assert all("family" in v for v in d.values())
    test = ModelRegistry.load(DEFAULT_PATH, profile="test", env={})
    assert test.build("dense_encoder").dim == 128
    default = ModelRegistry.load(DEFAULT_PATH, profile="default", env={})
    assert default.spec("verifier").family != default.spec("generator").family
    with pytest.raises(RegistryError):  # calibration block present but empty thresholds -> ok;
        default.thresholds("generator")  # generator has none at all -> refused
