main() {
    try{
    let correct = ArrayList<Bool>(
            allPrefixes("") == (ArrayList<String>()),
            allPrefixes("asdfgh") == (ArrayList<String>("a", "as", "asd", "asdf", "asdfg", "asdfgh")),
            allPrefixes("WWW") == (ArrayList<String>("W", "WW", "WWW"))
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