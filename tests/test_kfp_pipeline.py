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
