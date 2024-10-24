main() {
    try{
    let correct = ArrayList<Bool>(
            checkIfLastCharIsALetter("apple" ) == false,
            checkIfLastCharIsALetter("apple pi e" ) == true,
            checkIfLastCharIsALetter("eeeee" ) == false,
            checkIfLastCharIsALetter("A" ) == true,
            checkIfLastCharIsALetter("Pumpkin pie " ) == false,
            checkIfLastCharIsALetter("Pumpkin pie 1" ) == false,
            checkIfLastCharIsALetter("" ) == false,
            checkIfLastCharIsALetter("eeeee e " ) == false,
            checkIfLastCharIsALetter("apple pie" ) == false,
            checkIfLastCharIsALetter("apple pi e " ) == false
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