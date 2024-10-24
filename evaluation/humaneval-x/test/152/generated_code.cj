main() {
    try{
    let correct = ArrayList<Bool>(
            compare(ArrayList<Int64>(1, 2, 3, 4, 5, 1), ArrayList<Int64>(1, 2, 3, 4, 2, -2))==ArrayList<Int64>(0, 0, 0, 0, 3, 3),
            compare(ArrayList<Int64>(0,5,0,0,0,4), ArrayList<Int64>(4,1,1,0,0,-2))==ArrayList<Int64>(4,4,1,0,0,6),
            compare(ArrayList<Int64>(1, 2, 3, 4, 5, 1), ArrayList<Int64>(1, 2, 3, 4, 2, -2))==ArrayList<Int64>(0, 0, 0, 0, 3, 3),
            compare(ArrayList<Int64>(0, 0, 0, 0, 0, 0), ArrayList<Int64>(0, 0, 0, 0, 0, 0))==ArrayList<Int64>(0, 0, 0, 0, 0, 0),
            compare(ArrayList<Int64>(1, 2, 3), ArrayList<Int64>(-1, -2, -3))==ArrayList<Int64>(2, 4, 6),
            compare(ArrayList<Int64>(1, 2, 3, 5), ArrayList<Int64>(-1, 2, 3, 4))==ArrayList<Int64>(2, 0, 0, 1)
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