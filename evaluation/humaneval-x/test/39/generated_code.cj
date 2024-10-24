main() {
    try{
    let correct = ArrayList<Bool>(
            primeFib(1) == 2,
            primeFib(2) == 3,
            primeFib(3) == 5,
            primeFib(4) == 13,
            primeFib(5) == 89,
            primeFib(6) == 233,
            primeFib(7) == 1597,
            primeFib(8) == 28657,
            primeFib(9) == 514229,
            primeFib(10) == 433494437
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