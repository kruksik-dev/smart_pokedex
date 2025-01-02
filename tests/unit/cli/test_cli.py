import pyfiglet
from colorama import Fore, Style
from pytest import CaptureFixture

from smart_pokedex.cli.cli import display_prediction


def test_display_prediction(capsys: CaptureFixture[str]) -> None:
    """
    Test the display_prediction function to ensure it prints the correct output.
    """
    predicted_class = "Pikachu"
    confidence = 95.67

    expected_ascii_art = pyfiglet.figlet_format(predicted_class.upper())

    display_prediction(predicted_class, confidence)

    captured = capsys.readouterr()

    assert Fore.YELLOW + "=" * 50 in captured.out
    assert Fore.CYAN + expected_ascii_art in captured.out
    assert Fore.GREEN + f"Confidence Level: {confidence:.2f}%" in captured.out
    assert Fore.YELLOW + "=" * 50 + Style.RESET_ALL in captured.out
