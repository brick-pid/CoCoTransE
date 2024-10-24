main() {
    try{
    let correct = ArrayList<Bool>(
            willItFly(ArrayList<Int64>(3, 2, 3), 9),
            !willItFly(ArrayList<Int64>(1, 2), 5),
            willItFly(ArrayList<Int64>([3]), 5),
            !willItFly(ArrayList<Int64>(3, 2, 3), 1),
            !willItFly(ArrayList<Int64>(1, 2, 3), 6),
            willItFly(ArrayList<Int64>([5]), 5)
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