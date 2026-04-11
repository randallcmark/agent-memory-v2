from agent_memory_v2.classify_cli import build_parser


def test_classify_parser_accepts_semantic_flag():
    args = build_parser().parse_args(["--text", "I'm based in Edinburgh.", "--semantic"])

    assert args.text == "I'm based in Edinburgh."
    assert args.semantic is True


def test_classify_parser_accepts_extract_flag():
    args = build_parser().parse_args(["--text", "I'm based in Edinburgh.", "--extract"])

    assert args.text == "I'm based in Edinburgh."
    assert args.extract is True
