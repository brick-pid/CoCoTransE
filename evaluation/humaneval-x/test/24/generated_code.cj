main() {
    try{
    let correct = ArrayList<Bool>(
            largestDivisor(3) == 1,
            largestDivisor(7) == 1,
            largestDivisor(10) == 5,
            largestDivisor(100) == 50,
            largestDivisor(49) == 7
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