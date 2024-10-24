main() {
    try{
    let correct = ArrayList<Bool>(
            parseMusic("") == (ArrayList<Int64>()),
            parseMusic("o o o o") == (ArrayList<Int64>(4, 4, 4, 4)),
            parseMusic(".| .| .| .|") == (ArrayList<Int64>(1, 1, 1, 1)),
            parseMusic("o| o| .| .| o o o o") == (ArrayList<Int64>(2, 2, 1, 1, 4, 4, 4, 4)),
            parseMusic("o| .| o| .| o o| o o|") == (ArrayList<Int64>(2, 1, 2, 1, 4, 2, 4, 2))
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