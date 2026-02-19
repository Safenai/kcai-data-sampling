import sys
from unittest.mock import patch, MagicMock
import pytest
import yaml
from kcai_workspace.cli.main import main

def test_cli_help(capsys):
    """Test that the CLI shows help when no command is provided."""
    with patch.object(sys, 'argv', ['kcai-cli']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1
    
    captured = capsys.readouterr()
    assert "KCAI CLI: Unified entry point" in captured.out
    assert "Subcommands" in captured.out

@patch("kcai_workspace.cli.main.run_augment_pipeline")
@patch("builtins.open")
@patch("yaml.safe_load")
def test_cli_augment(mock_yaml_load, mock_open, mock_run_augment):
    """Test the 'augment' subcommand."""
    mock_yaml_load.return_value = {"dummy": "config"}
    mock_run_augment.return_value = "/path/to/output.parquet"
    
    with patch.object(sys, 'argv', ['kcai-cli', 'augment', '--config', 'test_config.yml']):
        main()
    
    mock_open.assert_called_once_with("test_config.yml", "r")
    mock_run_augment.assert_called_once_with({"dummy": "config"})

@patch("kcai_workspace.cli.main.run_adversarial_pipeline")
@patch("builtins.open")
@patch("yaml.safe_load")
def test_cli_adversarial(mock_yaml_load, mock_open, mock_run_adversarial):
    """Test the 'adversarial' subcommand."""
    mock_yaml_load.return_value = {"dummy": "adv_config"}
    mock_run_adversarial.return_value = "/path/to/adv_output.parquet"
    
    with patch.object(sys, 'argv', ['kcai-cli', 'adversarial', '--config', 'adv_config.yml']):
        main()
    
    mock_open.assert_called_once_with("adv_config.yml", "r")
    mock_run_adversarial.assert_called_once_with({"dummy": "adv_config"})

@patch("kcai_workspace.cli.main.run_inference_main")
def test_cli_inference(mock_run_inference):
    """Test the 'inference' subcommand."""
    # Test without --in-place
    with patch.object(sys, 'argv', ['kcai-cli', 'inference', '--config', 'inf_config.yml']):
        main()
    
    mock_run_inference.assert_called_once()
    # Check that sys.argv was correctly mocked during the call
    # Note: main() resets sys.argv, so we can't easily check sys.argv after the call
    # but we can verify that run_inference_main was called.

@patch("kcai_workspace.cli.main.run_inference_main")
def test_cli_inference_inplace(mock_run_inference):
    """Test the 'inference' subcommand with --in-place."""
    with patch.object(sys, 'argv', ['kcai-cli', 'inference', '--config', 'inf_config.yml', '--in-place']):
        main()
    
    mock_run_inference.assert_called_once()
