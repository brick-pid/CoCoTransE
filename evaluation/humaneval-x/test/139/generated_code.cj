main() {
    try{
    let correct = ArrayList<Bool>(
            specialFactorial(4) == 288,
            specialFactorial(5) == 34560,
            specialFactorial(7) == 125411328000,
            specialFactorial(1) == 1
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