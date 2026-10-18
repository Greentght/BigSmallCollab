import sys

import pytest

from experiments.distill import config_runner_impl, run_distill


PARSERS = [
    (run_distill.parse_args, []),
    (config_runner_impl.parse_args, ["--config", "fixture.yaml"]),
]


@pytest.mark.parametrize("parse_args, prefix", PARSERS)
def test_explicit_argv_ignores_standalone_whitespace_tokens(parse_args, prefix):
    args = parse_args([*prefix, "--seeds", " \t", "666"])
    assert args.seeds == [666]


@pytest.mark.parametrize("parse_args, prefix", PARSERS)
def test_argv_none_uses_process_argv_and_ignores_whitespace_tokens(
        monkeypatch, parse_args, prefix):
    monkeypatch.setattr(
        sys, "argv", ["run_distill.py", *prefix, "--seeds", "\n", "667"])
    args = parse_args()
    assert args.seeds == [667]


@pytest.mark.parametrize(
    "parse_args, prefix, following_args, field, expected",
    [
        (run_distill.parse_args, [], ["--temperature", "1.0"],
         "temperature", 1.0),
        (config_runner_impl.parse_args, ["--config", "fixture.yaml"],
         ["--gpu", "0"], "gpu", 0),
    ],
)
def test_seed_whitespace_before_following_option_does_not_consume_option(
        parse_args, prefix, following_args, field, expected):
    args = parse_args([
        *prefix, "--seeds", "0", "1", "2", " ", *following_args,
    ])

    assert args.seeds == [0, 1, 2]
    assert getattr(args, field) == expected


def test_normalization_preserves_empty_and_non_whitespace_tokens():
    original = [" /tmp/report file.csv ", "", "\t", "seed"]
    assert run_distill.normalize_cli_argv(original) == [
        " /tmp/report file.csv ", "", "seed"]


def test_path_arguments_containing_spaces_are_preserved():
    config_path = "/tmp/config files/loso distill.yaml"
    output_path = "/tmp/result files/teacher to student.csv"

    config_args = config_runner_impl.parse_args(["--config", config_path])
    legacy_args = run_distill.parse_args(["--out_csv", output_path])

    assert config_args.config == config_path
    assert legacy_args.out_csv == output_path


@pytest.mark.parametrize("parse_args, prefix", PARSERS)
@pytest.mark.parametrize("bad_seed", ["", "not-an-int"])
def test_empty_string_and_invalid_seed_are_not_discarded(
        parse_args, prefix, bad_seed):
    with pytest.raises(SystemExit):
        parse_args([*prefix, "--seeds", bad_seed])


@pytest.mark.parametrize("parse_args, prefix", PARSERS)
def test_missing_seed_value_still_fails_after_whitespace_filter(
        parse_args, prefix):
    with pytest.raises(SystemExit):
        parse_args([*prefix, "--seeds", "   "])


@pytest.mark.parametrize("parse_args, prefix", PARSERS)
def test_missing_seed_option_value_still_fails(parse_args, prefix):
    with pytest.raises(SystemExit):
        parse_args([*prefix, "--seeds"])
