main() {
    try{
    let correct = ArrayList<Bool>(
            factorize(2) == (ArrayList<Int64>([2])),
            factorize(4) == (ArrayList<Int64>(2, 2)),
            factorize(8) == (ArrayList<Int64>(2, 2, 2)),
            factorize(3 * 19) == (ArrayList<Int64>(3, 19)),
            factorize(3 * 19 * 3 * 19) == (ArrayList<Int64>(3, 3, 19, 19)),
            factorize(3 * 19 * 3 * 19 * 3 * 19) == (ArrayList<Int64>(3, 3, 3, 19, 19, 19)),
            factorize(3 * 19 * 19 * 19) == (ArrayList<Int64>(3, 19, 19, 19)),
            factorize(3 * 2 * 3) == (ArrayList<Int64>(2, 3, 3))
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