main() {
    try{
    let correct = ArrayList<Bool>(
            uniqueDigits(ArrayList<Int64>(15, 33, 1422, 1))==ArrayList<Int64>(1, 15, 33),
            uniqueDigits(ArrayList<Int64>(152, 323, 1422, 10))==ArrayList<Int64>(),
            uniqueDigits(ArrayList<Int64>(12345, 2033, 111, 151))==ArrayList<Int64>(111, 151),
            uniqueDigits(ArrayList<Int64>(135, 103, 31))==ArrayList<Int64>(31, 135)
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