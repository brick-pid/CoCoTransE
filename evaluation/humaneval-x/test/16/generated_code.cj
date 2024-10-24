main() {
    try{
    let correct = ArrayList<Bool>(
            countDistinctCharacters("") == 0,
            countDistinctCharacters("abcde") == 5,
            countDistinctCharacters("abcde" + "cade" + "CADE") == 5,
            countDistinctCharacters("aaaaAAAAaaaa") == 1,
            countDistinctCharacters("Jerry jERRY JeRRRY") == 5
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