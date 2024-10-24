main() {
    try{
    let correct = ArrayList<Bool>(
            stringSequence(0) == ("0"),
            stringSequence(3) == ("0 1 2 3"),
            stringSequence(10) == ("0 1 2 3 4 5 6 7 8 9 10")
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