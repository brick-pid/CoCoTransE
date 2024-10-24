
main() {
    try{
    let correct = ArrayList<Bool>(
            generateIntegers(2, 10)==ArrayList<Int64>(2, 4, 6, 8),
            generateIntegers(10, 2)==ArrayList<Int64>(2, 4, 6, 8),
            generateIntegers(132, 2)==ArrayList<Int64>(2, 4, 6, 8),
            generateIntegers(17, 89)==ArrayList<Int64>()
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