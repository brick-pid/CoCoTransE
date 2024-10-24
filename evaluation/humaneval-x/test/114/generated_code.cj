main() {
    try{
    let correct = ArrayList<Bool>(
            minSubArraySum(ArrayList<Int64>(2, 3, 4, 1, 2, 4)) == 1,
            minSubArraySum(ArrayList<Int64>(-1, -2, -3)) == -6,
            minSubArraySum(ArrayList<Int64>(-1, -2, -3, 2, -10)) == -14,
            minSubArraySum(ArrayList<Int64>([-999999999])) == -999999999,
            minSubArraySum(ArrayList<Int64>(0, 10, 20, 1000000)) == 0,
            minSubArraySum(ArrayList<Int64>(-1, -2, -3, 10, -5)) == -6,
            minSubArraySum(ArrayList<Int64>(100, -1, -2, -3, 10, -5)) == -6,
            minSubArraySum(ArrayList<Int64>(10, 11, 13, 8, 3, 4)) == 3,
            minSubArraySum(ArrayList<Int64>(100, -33, 32, -1, 0, -2)) == -33,
            minSubArraySum(ArrayList<Int64>([-10])) == -10,
            minSubArraySum(ArrayList<Int64>([7])) == 7,
            minSubArraySum(ArrayList<Int64>(1, -1)) == -1
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