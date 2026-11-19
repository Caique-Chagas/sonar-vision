import httpx
import pytest

from sonar_vision.br.curation import BoundedSourceClient, CollectionBudget, CollectionStopped


@pytest.mark.parametrize("status", [403, 429])
def test_restriction_stops_across_instances_without_retry(tmp_path, status):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(status, headers={"Retry-After": "600"})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    source = BoundedSourceClient(tmp_path, client)
    with pytest.raises(CollectionStopped):
        source.fetch("https://commons.wikimedia.org/api", max_bytes=100)
    with pytest.raises(CollectionStopped):
        BoundedSourceClient(tmp_path, client).fetch(
            "https://commons.wikimedia.org/other", max_bytes=100
        )
    assert len(calls) == 1
    assert "600" in source.stop.read_text()


def test_byte_and_request_budgets_persist(tmp_path):
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"largebody"))
    )
    with pytest.raises(CollectionBudget):
        BoundedSourceClient(tmp_path, client, max_requests=1).fetch(
            "https://upload.wikimedia.org/a", max_bytes=2
        )
    with pytest.raises(CollectionBudget):
        BoundedSourceClient(tmp_path, client, max_requests=1).fetch(
            "https://upload.wikimedia.org/b", max_bytes=100
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://commons.wikimedia.org/a",
        "https://evil.example/a",
        "https://user:pass@commons.wikimedia.org/a",
    ],
)
def test_unapproved_urls_never_reach_transport(tmp_path, url):
    def forbidden(request):
        raise AssertionError("must not request")

    source = BoundedSourceClient(tmp_path, httpx.Client(transport=httpx.MockTransport(forbidden)))
    with pytest.raises(ValueError):
        source.fetch(url, max_bytes=100)


def test_single_resume_honors_cooldown_and_preserves_429(tmp_path):
    import os
    import time

    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "1"})

    source = BoundedSourceClient(tmp_path, httpx.Client(transport=httpx.MockTransport(handle)))
    with pytest.raises(CollectionStopped):
        source.fetch("https://upload.wikimedia.org/a", max_bytes=100)
    with pytest.raises(CollectionStopped):
        source.resume_once_after_retry()
    old = time.time() - 120
    os.utime(source.stop, (old, old))
    os.utime(source.ledger, (old, old))
    source.resume_once_after_retry()
    assert (tmp_path / "http-stop-before-single-resume.json").exists()
    with pytest.raises(CollectionStopped):
        source.fetch("https://upload.wikimedia.org/a", max_bytes=100)
    with pytest.raises(CollectionStopped):
        source.resume_once_after_retry()
    assert len(calls) == 2


def test_documented_photographer_aliases_do_not_create_new_capture_groups():
    from sonar_vision.br.curation import canonical_author

    assert canonical_author("Mariordo (Mario Roberto Durán Ortiz)") == canonical_author(
        "Mario Roberto Duran Ortiz Mariordo"
    )
    assert canonical_author("Rafael Delazari from Belo Horizonte, Brazil") == canonical_author(
        "Rafael Delazari"
    )
    assert canonical_author("Eliezer Santos") != canonical_author("Asprilla~ptwiki")


def test_clients_created_before_a_request_share_the_persisted_budget(tmp_path):
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"ok"))
    )
    first = BoundedSourceClient(tmp_path, client, max_requests=1)
    stale = BoundedSourceClient(tmp_path, client, max_requests=1)
    assert first.fetch("https://commons.wikimedia.org/a", max_bytes=10) == b"ok"
    with pytest.raises(CollectionBudget):
        stale.fetch("https://commons.wikimedia.org/b", max_bytes=10)


def sample_panel(index):
    return {
        "reviewed": True,
        "rights_verified": True,
        "image_sha256": f"hash-{index}",
        "width": 1600,
        "height": 1200,
        "group_id": f"group-{index}",
        "boxes": [{"approved": True, "xyxy": [100, 100, 700, 200]}],
        "exhaustive_annotation": True,
        "expected_code": "5109-10",
        "windshield_integrated": True,
    }


def test_many_new_capture_groups_do_not_compensate_for_missing_small_panels():
    from sonar_vision.br.curation import assess_targeted_coverage

    result = assess_targeted_coverage([sample_panel(i) for i in range(12)])
    assert not result["coverage_passed"]
    assert result["counts"]["positive_groups"] == 12
    assert result["counts"]["small_panels"] == 0


def test_duplicates_and_unreviewed_licensing_rejected_before_coverage():
    from sonar_vision.br.curation import assess_targeted_coverage

    with pytest.raises(ValueError, match="duplicate"):
        assess_targeted_coverage([sample_panel(0), sample_panel(0)])
    bad = sample_panel(0)
    bad["rights_verified"] = False
    with pytest.raises(ValueError, match="rights"):
        assess_targeted_coverage([bad])


def test_negatives_never_count_as_positive_capture_groups():
    from sonar_vision.br.curation import assess_targeted_coverage

    negative = sample_panel(1)
    negative.update(boxes=[], confirmed_negative=True)
    result = assess_targeted_coverage([sample_panel(0), negative])
    assert result["counts"]["positive_groups"] == 1
