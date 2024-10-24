main() {
    try{
    let correct = ArrayList<Bool>(
            sumSquares(ArrayList<Float64>(1, 2, 3)) == 14,
            sumSquares(ArrayList<Float64>(1.0, 2, 3)) == 14,
            sumSquares(ArrayList<Float64>(1, 3, 5, 7)) == 84,
            sumSquares(ArrayList<Float64>(1.4, 4.2, 0)) == 29,
            sumSquares(ArrayList<Float64>(-2.4, 1, 1)) == 6,
            sumSquares(ArrayList<Float64>(100, 1, 15, 2)) == 10230,
            sumSquares(ArrayList<Float64>(10000, 10000)) == 200000000,
            sumSquares(ArrayList<Float64>(-1.4, 4.6, 6.3)) == 75,
            sumSquares(ArrayList<Float64>(-1.4, 17.9, 18.9, 19.9)) == 1086,
            sumSquares(ArrayList<Float64>([0.0])) == 0,
            sumSquares(ArrayList<Float64>([-1])) == 1,
            sumSquares(ArrayList<Float64>(-1, 1, 0)) == 2
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