main() {
    try{
    let correct = ArrayList<Bool>(
            sameChars("eabcdzzzz", "dddzzzzzzzddeddabc"),
            sameChars("abcd", "dddddddabc"),
            sameChars("dddddddabc", "abcd"),
            !sameChars("eabcd", "dddddddabc"),
            !sameChars("abcd", "dddddddabcf"),
            !sameChars("eabcdzzzz", "dddzzzzzzzddddabc"),
            !sameChars("aabb", "aaccc")
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