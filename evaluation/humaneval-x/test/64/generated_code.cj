main() {
    try{
    let correct = ArrayList<Bool>(
            vowelsCount("abcde") == 2,
            vowelsCount("Alone") == 3,
            vowelsCount("key") == 2,
            vowelsCount("bye") == 1,
            vowelsCount("keY") == 2,
            vowelsCount("bYe") == 1,
            vowelsCount("ACEDY") == 3
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