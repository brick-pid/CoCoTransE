main() {
    try{
    let correct = ArrayList<Bool>(
            splitWords("Hello world!") == ArrayList<Int64>("Hello", "world!"),
            splitWords("Hello,world!") == ArrayList<Int64>("Hello", "world!"),
            splitWords("Hello world,!") == ArrayList<Int64>("Hello", "world,!"),
            splitWords("Hello,Hello,world !") == ArrayList<Int64>("Hello,Hello,world", "!"),
            splitWords("abcdef") == 3,
            splitWords("aaabb") == 2,
            splitWords("aaaBb") == 1,
            splitWords("") == 0
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