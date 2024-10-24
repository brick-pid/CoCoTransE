main() {
    try{
    let correct = ArrayList<Bool>(
            digits(5) == 5,
            digits(54) == 5,
            digits(120) == 1,
            digits(5014) == 5,
            digits(98765) == 315,
            digits(5576543) == 2625
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