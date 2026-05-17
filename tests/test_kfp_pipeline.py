import yaml


def test_pipeline_compiles_without_error(tmp_path):
    from kfp import compiler
    from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=str(out),
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_pipeline_yaml_has_correct_components(tmp_path):
    from kfp import compiler
    from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=str(out),
    )
    spec = yaml.safe_load(out.read_text())
    executors = spec.get("deploymentSpec", {}).get("executors", {})
    assert len(executors) >= 8


def test_pipeline_base_image_is_correct(tmp_path):
    from kfp import compiler
    from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=str(out),
    )
    spec = yaml.safe_load(out.read_text())
    executors = spec.get("deploymentSpec", {}).get("executors", {})
    for name, executor in executors.items():
        image = executor.get("container", {}).get("image", "")
        assert "smart-ecommerce-pipeline:local" not in image, (
            f"Component {name} still uses old base image: {image}"
        )
        assert image == "smart-ecommerce-pipeline-v2-app:latest", (
            f"Component {name} has unexpected image: {image}"
        )


def test_pipeline_no_sys_path_append(tmp_path):
    from kfp import compiler
    from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=str(out),
    )
    content = out.read_text()
    assert "sys.path.append" not in content


def test_pipeline_has_data_dir_parameter(tmp_path):
    from kfp import compiler
    from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=str(out),
    )
    spec = yaml.safe_load(out.read_text())
    root = spec.get("root", {})
    input_defs = root.get("inputDefinitions", {}).get("parameters", {})
    assert "data_dir" in input_defs, f"data_dir not found. Got: {list(input_defs)}"


def test_preprocess_caching_disabled(tmp_path):
    from kfp import compiler
    from src.pipeline.kubeflow_pipeline import smart_ecommerce_pipeline

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=str(out),
    )
    spec = yaml.safe_load(out.read_text())
    tasks = spec.get("root", {}).get("dag", {}).get("tasks", {})
    preprocess_task = None
    for name, task in tasks.items():
        if "preprocess" in name.lower():
            preprocess_task = task
            break
    assert preprocess_task is not None, "preprocess task not found"
    caching = preprocess_task.get("cachingOptions", {})
    assert caching.get("enableCache") is not True, (
        f"preprocess_op should have caching disabled, got: {caching}"
    )
