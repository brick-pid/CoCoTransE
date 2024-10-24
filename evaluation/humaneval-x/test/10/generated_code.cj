func isPalindrome(string: String): Bool {
    var i = 0
    var j = string.size - 1
    while (i < j) {
        if (string[i] != string[j]) {
            return false
        }
        i = i + 1
        j = j - 1
    }
    true
}

main() {
    try{
    let correct = ArrayList<Bool>(
            (makePalindrome("") == ""),
            (makePalindrome("x") == "x"),
            (makePalindrome("xyz") == "xyzyx"),
            (makePalindrome("xyx") == "xyx"),
            (makePalindrome("jerry") == "jerryrrej")
    );
if (correct.contains(false)) {
                print(correct)
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}