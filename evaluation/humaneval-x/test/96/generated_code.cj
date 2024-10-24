main() {
    try{
    let correct = ArrayList<Bool>(
            countUpTo(5) == ArrayList<Int64>(2, 3),
            countUpTo(6) == ArrayList<Int64>(2, 3, 5),
            countUpTo(7) == ArrayList<Int64>(2, 3, 5),
            countUpTo(10) == ArrayList<Int64>(2, 3, 5, 7),
            countUpTo(0) == ArrayList<Int64>(),
            countUpTo(22) == ArrayList<Int64>(2, 3, 5, 7, 11, 13, 17, 19),
            countUpTo(1) == ArrayList<Int64>(),
            countUpTo(18) == ArrayList<Int64>(2, 3, 5, 7, 11, 13, 17),
            countUpTo(47) == ArrayList<Int64>(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43),
            countUpTo(101) == ArrayList<Int64>(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
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