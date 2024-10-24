main() {
    try{
    let correct = ArrayList<Bool>(
            !belowZero(ArrayList<Int64>()),
            !belowZero(ArrayList<Int64>(1, 2, -3, 1, 2, -3)),
            belowZero(ArrayList<Int64>(1, 2, -4, 5, 6)),
            !belowZero(ArrayList<Int64>(1, -1, 2, -2, 5, -5, 4, -4)),
            belowZero(ArrayList<Int64>(1, -1, 2, -2, 5, -5, 4, -5)),
            belowZero(ArrayList<Int64>(1, -2, 2, -2, 5, -5, 4, -4))
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