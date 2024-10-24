main() {
    try{
    let correct = ArrayList<Bool>(
            howManyTimes("", "x") == 0,
            howManyTimes("xyxyxyx", "x") == 4,
            howManyTimes("cacacacac", "cac") == 4,
            howManyTimes("john doe", "john") == 1
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