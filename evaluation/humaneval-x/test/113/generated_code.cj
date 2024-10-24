main() {
    try{
    let correct = ArrayList<Bool>(
            oddCount(ArrayList<String>("1234567" ))==ArrayList<String>("the number of odd elements 4n the str4ng 4 of the 4nput." ),
            oddCount(ArrayList<String>("3", "11111111" ))==ArrayList<String>("the number of odd elements 1n the str1ng 1 of the 1nput.", "the number of odd elements 8n the str8ng 8 of the 8nput." ),
            oddCount(ArrayList<String>("271", "137", "314" ))==ArrayList<String>(
                    "the number of odd elements 2n the str2ng 2 of the 2nput.",
                    "the number of odd elements 3n the str3ng 3 of the 3nput.",
                    "the number of odd elements 2n the str2ng 2 of the 2nput."
            )
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