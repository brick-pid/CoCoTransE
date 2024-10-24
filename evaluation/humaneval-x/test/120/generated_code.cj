main() {
    try{
    let correct = ArrayList<Bool>(
            maximum(ArrayList<Int64>(-3, -4, 5), 3)==ArrayList<Int64>(-4, -3, 5),
            maximum(ArrayList<Int64>(4, -4, 4), 2)==ArrayList<Int64>(4, 4),
            maximum(ArrayList<Int64>(-3, 2, 1, 2, -1, -2, 1), 1)==ArrayList<Int64>([2]),
            maximum(ArrayList<Int64>(123, -123, 20, 0 , 1, 2, -3), 3)==ArrayList<Int64>(2, 20, 123),
            maximum(ArrayList<Int64>(-123, 20, 0 , 1, 2, -3), 4)==ArrayList<Int64>(0, 1, 2, 20),
            maximum(ArrayList<Int64>(5, 15, 0, 3, -13, -8, 0), 7)==ArrayList<Int64>(-13, -8, 0, 0, 3, 5, 15),
            maximum(ArrayList<Int64>(-1, 0, 2, 5, 3, -10), 2)==ArrayList<Int64>(3, 5),
            maximum(ArrayList<Int64>(1, 0, 5, -7), 1)==ArrayList<Int64>([5]),
            maximum(ArrayList<Int64>(4, -4), 2)==ArrayList<Int64>(-4, 4),
            maximum(ArrayList<Int64>(-10, 10), 2)==ArrayList<Int64>(-10, 10)
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