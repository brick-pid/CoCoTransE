main() {
    try{
    let correct = ArrayList<Bool>(
            isPalindrome(""),
            isPalindrome("aba"),
            isPalindrome("aaaaa"),
            !isPalindrome("zbcd"),
            isPalindrome("xywyx"),
            !isPalindrome("xywyz"),
            !isPalindrome("xywzx")
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