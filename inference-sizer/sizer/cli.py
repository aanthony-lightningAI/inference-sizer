"""CLI: python -m sizer.cli  or  python -m sizer.cli request.json"""

import json
import sys

from sizer.engine import SizeRequest, size


def main() -> None:
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            req = SizeRequest.model_validate(json.load(f))
    else:
        req = SizeRequest()
    json.dump(size(req), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
