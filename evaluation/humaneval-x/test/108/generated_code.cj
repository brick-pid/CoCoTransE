main() {
    try{
    let correct = ArrayList<Bool>(
            countNums(ArrayList<Int64>()) == 0,
            countNums(ArrayList<Int64>(-1, -2, 0)) == 0,
            countNums(ArrayList<Int64>(1, 1, 2, -2, 3, 4, 5)) == 6,
            countNums(ArrayList<Int64>(1, 6, 9, -6, 0, 1, 5)) == 5,
            countNums(ArrayList<Int64>(1, 100, 98, -7, 1, -1)) == 4,
            countNums(ArrayList<Int64>(12, 23, 34, -45, -56, 0)) == 5,
            countNums(ArrayList<Int64>(-0, Int64(math.pow(1.0, 0)))) == 1,
            countNums(ArrayList<Int64>([1])) == 1
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