main() {
    try{
    let correct = ArrayList<Bool>(
            !isMultiplyPrime(5),
            isMultiplyPrime(30),
            isMultiplyPrime(8),
            !isMultiplyPrime(10),
            isMultiplyPrime(125),
            isMultiplyPrime(3 * 5 * 7),
            !isMultiplyPrime(3 * 6 * 7),
            !isMultiplyPrime(9 * 9 * 9),
            !isMultiplyPrime(11 * 9 * 9),
            isMultiplyPrime(11 * 13 * 7)
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