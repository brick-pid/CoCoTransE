main() {
    try{
    let correct = ArrayList<Bool>(
            derivative(ArrayList<Int64>(3, 1, 2, 4, 5)) == (ArrayList<Int64>(1, 4, 12, 20)),
            derivative(ArrayList<Int64>(1, 2, 3)) == (ArrayList<Int64>(2, 6)),
            derivative(ArrayList<Int64>(3, 2, 1)) == (ArrayList<Int64>(2, 2)),
            derivative(ArrayList<Int64>(3, 2, 1, 0, 4)) == (ArrayList<Int64>(2, 2, 0, 16)),
            derivative(ArrayList<Int64>([1])) == (ArrayList<Int64>())
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