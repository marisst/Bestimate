import sys
from translate.tokens_module import count_tokens

if len(sys.argv) != 2:
    print("Please select one dataset")
    sys.exit();

count_tokens(sys.argv[1])