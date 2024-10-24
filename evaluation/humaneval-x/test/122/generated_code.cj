main() {
    try{
    let correct = ArrayList<Bool>(
            addElements(ArrayList<Int64>(1, -2, -3, 41, 57, 76, 87, 88, 99), 3) == -4,
            addElements(ArrayList<Int64>(111, 121, 3, 4000, 5, 6), 2) == 0,
            addElements(ArrayList<Int64>(11, 21, 3, 90, 5, 6, 7, 8, 9), 4) == 125,
            addElements(ArrayList<Int64>(111, 21, 3, 4000, 5, 6, 7, 8, 9), 4) == 24,
            addElements(ArrayList<Int64>([1]), 1) == 1
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