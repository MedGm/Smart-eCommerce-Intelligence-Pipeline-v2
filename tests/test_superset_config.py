from pathlib import Path


def test_superset_config_exists():
    assert Path("superset/superset_config.py").exists(), (
        "superset/superset_config.py not found"
    )


def test_superset_config_has_required_keys():
    content = Path("superset/superset_config.py").read_text()
    assert "SECRET_KEY" in content
    assert "SQLALCHEMY_DATABASE_URI" in content


def test_dockerfile_superset_exists():
    assert Path("Dockerfile.superset").exists(), "Dockerfile.superset not found"


def test_dockerfile_superset_installs_duckdb_engine():
    content = Path("Dockerfile.superset").read_text()
    assert "duckdb-engine" in content


def test_docker_compose_has_superset_service():
    import yaml
    spec = yaml.safe_load(Path("docker-compose.yml").read_text())
    services = spec.get("services", {})
    assert "superset" in services, f"superset service not found. Got: {list(services)}"
    assert "superset-init" in services, "superset-init service not found"


def test_superset_service_has_correct_profile():
    import yaml
    spec = yaml.safe_load(Path("docker-compose.yml").read_text())
    superset = spec["services"]["superset"]
    profiles = superset.get("profiles", [])
    assert "superset" in profiles


def test_superset_data_volume_mounted_readonly():
    import yaml
    spec = yaml.safe_load(Path("docker-compose.yml").read_text())
    superset = spec["services"]["superset"]
    volumes = superset.get("volumes", [])
    data_mounts = [v for v in volumes if "data" in str(v)]
    assert any(":ro" in str(v) for v in data_mounts), (
        "data/ not mounted read-only in superset service"
    )
