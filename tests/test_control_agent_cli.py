from control_agent import _build_parser


def test_prepare_skip_llm1_defaults_to_false():
    args = _build_parser().parse_args(["prepare", "--config", "config.yaml"])
    assert args.skip_llm1 is False


def test_prepare_skip_llm1_flag_sets_true():
    args = _build_parser().parse_args(["prepare", "--config", "config.yaml", "--skip_llm1"])
    assert args.skip_llm1 is True
