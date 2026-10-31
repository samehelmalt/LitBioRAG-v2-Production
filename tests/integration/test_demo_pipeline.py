from scripts import demo_pipeline


def test_demo_runs_end_to_end_on_fixture_bundle(fixture_bundle):
    out = demo_pipeline.run(fixture_bundle, "What does MAP3K3 do in NF-kappaB signalling?", "test")
    assert out["evidence"][0]["citation"] == "PMID:900001"
    assert out["answer"]["abstained"] is False
    v = out["verification"]
    assert v["n_claims"] >= 1 and v["entailment_rate"] == 1.0
    assert not v["fabricated_citations"]
    assert set(out["models"]) >= {"dense_encoder", "reranker", "generator", "verifier"}
    assert out["decision"]["action"] in ("ANSWER", "ANSWER_WITH_CAUTION")
    assert out["decision"]["rule"] != "R1-scope"  # in scope
    assert out["decision"]["rule"] in ("R10-repair", "R12-confidence")
    assert "UNCALIBRATED" in out["gate_calibrated_on"]
    assert all(k in out["timings_ms"] for k in ("retrieval", "generation", "verification"))


def test_demo_abstains_on_unanswerable_question(fixture_bundle):
    out = demo_pipeline.run(fixture_bundle, "Does aspirin prevent migraine in adolescents?", "test")
    # evidence exists (filler documents) but nothing overlaps: the template generator abstains
    assert out["answer"]["abstained"] is True
    assert out["verification"]["n_claims"] == 0
    assert out["decision"]["action"] == "ABSTAIN"
    assert out["decision"]["rule"] == "R2-generator-abstained"
