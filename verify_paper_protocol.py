"""Static checks for the ICASSP paper protocol defaults."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "train_vgms.py"

EXPECTED_DEFAULTS = {
    "epochs": 80,
    "patience": 10,
    "lr": 5e-5,
    "min_lr": 1e-6,
    "loss": "focal",
    "focal_gamma": 2.0,
    "test_size": 0.3,
    "val_size": 2.0 / 7.0,
    "seed": 42,
    "n_bootstraps": 2000,
    "max_patches_per_case": 1024,
    "align_topk": 64,
    "align_temperature": 1.0,
    "spatial_gamma": 0.25,
    "spatial_sigma": 8.0,
    "patch_size": 250,
    "morphology_mode": "macenko",
    "text_encoder_type": "clinicalbert",
}


def literal_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -literal_value(node.operand)
    if isinstance(node, ast.BinOp):
        left = literal_value(node.left)
        right = literal_value(node.right)
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Mult):
            return left * right
    raise ValueError(f"Unsupported default expression: {ast.dump(node)}")


def parser_defaults(tree):
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                defaults[flag[2:]] = literal_value(keyword.value)
    return defaults


def query_count(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "DEFAULT_QUERIES" for target in node.targets):
                return len(node.value.elts)
    return 0


def main():
    source = TRAIN_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaults = parser_defaults(tree)
    errors = []

    if query_count(tree) != 6:
        errors.append("DEFAULT_QUERIES must contain exactly six paper queries")
    for name, expected in EXPECTED_DEFAULTS.items():
        actual = defaults.get(name)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-12:
                errors.append(f"--{name}: expected {expected!r}, found {actual!r}")
        elif actual != expected:
            errors.append(f"--{name}: expected {expected!r}, found {actual!r}")

    required_snippets = {
        "patient-group splitting": 'groupby("case_id")',
        "VGMS model class": "class VGMS(nn.Module)",
        "IHC-to-H&E feature cosine": "F.normalize(ihc_features, dim=1) @ F.normalize(he_features, dim=1).t()",
        "hematoxylin SSIM": "def _pairwise_ssim",
        "spiral candidate traversal": "order = spiral_candidate_order(he_yx, origin)",
        "distance-aware alignment": "s_base - self.spatial_gamma * spatial_penalty",
        "Top-64 selection": "torch.argsort(",
        "soft H&E aggregation": "torch.einsum(",
        "patient-cluster bootstrap": "group_members = [np.flatnonzero",
        "query-to-visual cross-attention": "query_tokens @ visual_keys.t()",
        "no-query gated MIL baseline": "self.mil_attention_w",
        "three-layer gated classifier": "class ThreeLayerGatedMLP",
        "cosine learning-rate decay": "optim.lr_scheduler.CosineAnnealingLR",
        "test-free hyperparameter selection": "args.skip_test_evaluation",
    }
    for label, snippet in required_snippets.items():
        if snippet not in source:
            errors.append(f"Missing {label}: {snippet}")

    lambda_selector = (ROOT / "select_alignment_lambda.py").read_text(encoding="utf-8")
    for snippet in ["validation_auc", "--skip_test_evaluation", "test_cohorts_used_for_selection"]:
        if snippet not in lambda_selector:
            errors.append(f"Alignment-lambda selector is missing: {snippet}")

    if errors:
        print("Paper protocol check failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Paper protocol check passed.")


if __name__ == "__main__":
    main()
