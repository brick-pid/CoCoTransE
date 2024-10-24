main() {
    try{
    let correct = ArrayList<Bool>(
            simplify("1/5", "5/1") == true,
            simplify("1/6", "2/1") == false,
            simplify("5/1", "3/1") == true,
            simplify("7/10", "10/2") == false,
            simplify("2/10", "50/10") == true,
            simplify("7/2", "4/2") == true,
            simplify("11/6", "6/1") == true,
            simplify("2/3", "5/2") == false,
            simplify("5/2", "3/5") == false,
            simplify("2/4", "8/4") == true,
            simplify("2/4", "4/2") == true,
            simplify("1/5", "5/1") == true,
            simplify("1/5", "1/5") == false
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