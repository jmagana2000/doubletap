from pathlib import Path


def recognize_text(image_path: Path) -> list[tuple[str, float]]:
    """Read text lines from an image with Apple's Vision framework.

    Returns (text, confidence) per detected line, top to bottom. Requires the
    `ocr` extra (pyobjc) and macOS; imports lazily so everything downstream
    stays testable by mocking this one function.
    """
    try:
        from Foundation import NSURL
        from Vision import (
            VNImageRequestHandler,
            VNRecognizeTextRequest,
            VNRequestTextRecognitionLevelAccurate,
        )
    except ImportError as e:
        raise RuntimeError(
            "Photo import needs the Apple Vision framework: pip install 'doubletap[ocr]' (macOS only)"
        ) from e

    url = NSURL.fileURLWithPath_(str(Path(image_path).resolve()))
    handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    request = VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
    # card names are not dictionary words; language correction mangles them
    request.setUsesLanguageCorrection_(False)

    success, error = handler.performRequests_error_([request], None)
    if not success:
        raise RuntimeError(f"Vision OCR failed: {error}")

    lines = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if candidates:
            text = candidates[0]
            lines.append((str(text.string()), float(text.confidence())))
    return lines
