import java.util.*;
import java.io.*;

public class Tree {
	static int p = 0, n = 0;//total number of positives and negatives of the training data
	static double ent = 0;//entropy of table
	
	public static void main(String [] args)throws IOException{
		Scanner sc = new Scanner(System.in);
		File check;//used to ensure user enters a valid training data file
		String path = "C:\\Users\\Zal\\Desktop\\";//default directory
		do {
			System.out.println("Enter training data file name (must be a csv file). If no path is mentioned default is C:\\Users\\Zal\\Desktop");
			String sTemp = sc.nextLine();
			if(sTemp.contains("\\")) {//if there is some \ then user has probably given a path
				path = sTemp;
			}
			else {
				path += sTemp;
			}
			if(!path.endsWith(".csv")) {//adding .csv if they have not mentioned it
				path+=".csv";
			}
			check = new File(path);
			if(!check.exists()) {
				System.out.println("Incorrect filename");
			}
		}
		while(!check.exists());
		
		BufferedReader br = new BufferedReader(new FileReader(check));
		String full = "";
		String s;
		int numRows = 0;
		int numCols = 0;
		//accepting entire file as a string, separating columns with _
		while((s = br.readLine()) != null){
			full+=s;
			full+="_";
			numCols++;
		}
		
		char c;
		int pos =0;
		while((c=full.charAt(pos)) != '_'){
			pos++;
			// , means next attribute
			// storing each attribute as a row in table(so that its values will be in the same array), so we count number of rows
			if(c==','){
				numRows++;
			}
		}
		numRows++;
		//last attribute will not have a , after it
		
		pos = 0;
		int sPos = 0;
		//sPos for start of word, pos for the end
		
		String [][] tab = new String[numRows][numCols];
		
		for(int i = 0; i < numCols; i++){
			for(int j = 0; j < numRows; j++){
				while((c=full.charAt(pos)) != ',' && (c=full.charAt(pos)) != '_'){
					// , or _ means end of word
					pos++;
				}
				tab[j][i] = full.substring(sPos,pos);
				sPos = pos+1;
				pos++;
			}
		}
		br.close();
		
		int targetIndex = tab.length-1;
		//target attribute index
		
		int [][] tabVals = convert(tab);
		//converting attribute values to integer values
		
		findPN(tabVals[targetIndex]);
		//assigning values to p and n (total number of positives and negatives)
		
		ent = entropy(p,n);
		//I of table
		
		double [] rem = new double[tabVals.length-1];
		//gain of all attributes except target, so length - 1
		
		for(int i = 0; i < tabVals.length-1; i++) {
			rem[i] = findRem(tabVals[i], tabVals[targetIndex]);
			//finding remainder for all attributes and storing in array
		}
		
		int [] ogPos = new int[rem.length];//storing original order of attributes to write to file later
		for(int i = 0; i < ogPos.length; i++) {
			ogPos[i] = i;
			//positions from 0 to n (n is number of attributes)
		}
		
		//now sorting tabVals, tab and rem according to rem, in descending order (higher remainder = lower gain)
		for(int i = 0; i < rem.length; i++) {
			for(int j = 0; j < rem.length; j++) {
				if(rem[i] < rem[j]) {
					//swap temp
					double temp = rem[j];
					rem[j] = rem[i];
					rem[i] = temp;
					
					//swap tab
					String [] tabTemp = tab[i];
					tab[i] = tab[j];
					tab[j] = tabTemp;
					
					//swap tabVals
					int [] tabValsTemp = tabVals[i];
					tabVals[i] = tabVals[j];
					tabVals[j] = tabValsTemp;
					
					//swapping positions to keep track of where each thing was
					int posTemp = ogPos[i];
					ogPos[i] = ogPos[j];
					ogPos[j] = posTemp;
				}
			}
		}
		
		String [] newValues = new String[tabVals.length-1];	//holds values of attribute for row to classify	
		//length - 1 since there is no classification yet
		
		//accepting values from user to predict
		System.out.println("Enter values to predict");
		for(int i = 0; i < newValues.length; i++) {
			System.out.println(tab[i][0] + " : ");
			newValues[i] = sc.nextLine();
		}
		int [] newVal = turnInt(newValues,tab, tabVals);
		//converting string values to their equivalent int values(using training data values)
		
		int [] target = tabVals[targetIndex];
		String [] tVals = tab[targetIndex];
		//storing target row as a separate array(one for int values, one for string)

		int [] max = conMax(tabVals);
		//stores value of each attribute that occurs max times(to implement maxFunction)
	
		newVal = conv(newVal,max);
		//converting test data so that if any value is present that does not occur in training data, it gets converted to the value of that attribute that occurs most
		//implements max function
		
		boolean [] arr = new boolean[tabVals[0].length];
		//to see if we should check the value of attribute at this position or not(used in prediction)
		for(int i = 0; i < arr.length; i++) {
			arr[i] = true;
		}//initialised to 0 since for first attribute we check all values
		
		String classify = predict(tabVals, newVal, 0, arr, target, tVals,"");
		System.out.println(tab[targetIndex][0] + " : " + classify);
		//final output is the classification
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(path));
		StringTokenizer tok = new StringTokenizer(full,"_");
		while(tok.hasMoreTokens()) {
			bw.write(tok.nextToken()+"\n");
		}
		
		//sorting newValues according to ogPos values to get them in the right order to write to file
		for(int i = 0; i < ogPos.length; i++) {
			for(int j = 0; j < ogPos.length; j++) {
				if(ogPos[i] < ogPos[j]) {
					//sorting ogPos
					int posTemp = ogPos[i];
					ogPos[i] = ogPos[j];
					ogPos[j] = posTemp;
					
					//sorting newValues
					String temp = newValues[i];
					newValues[i] = newValues[j];
					newValues[j] = temp;
				}
			}
		}
		
		
		for(int i = 0; i < newValues.length; i++) {
			bw.write(newValues[i] + ",");
		}
		bw.write(classify);
		bw.close();
	}
	
	public static int [][] convert(String [][] arr){
		
		int [][] tab = new int[arr.length][arr[0].length-1];
		//array of breadth 1 less than original to exclude titles
		
		ArrayList<String> temp = new ArrayList<String>();
		//stores values used (for each attribute)
		
		for(int i = 0; i < arr.length; i++) {
			temp.clear(); //clearing for next row
			int num = 0; //holds number of attributes 
			
			for(int j = 1; j < arr[0].length; j++) {
				if(temp.contains((String)arr[i][j])) {
					//if already existent value then replace with its position in temp
					tab[i][j-1] = temp.indexOf(arr[i][j]);
				}
				else {
					//else replace it with num, place it in temp and increment num
					temp.add(arr[i][j]);
					tab[i][j-1] = num++;
				}
			}
		}
		
		return tab;	
	}
	
	public static void findPN(int [] arr) {
		for(int j = 0; j < arr.length; j++) {
			if(arr[j] == 0) {
				p++;
				//0 is target value
			}
			else {
				n++;
				//1 is not target value
			}
		}
	}
	
	public static double entropy(int p, int n) {
		double i = 0;
		
		//formula I= -(p+n)log2(p/p+n) - (n/n+p)log2(n/p+n)
		if(p != 0) {
			i = (-1)*(double)(p)/(double)(p+n);
			i *= Math.log10((double)p/(double)(p+n))/Math.log10(2);
		}
		if(n == 0) {
			return i;
		}
		i -= ((double)(n)/(double)(p+n))*(Math.log10((double)n/(double)(p+n))/Math.log10(2)); 
		//typecasting p and n to double since integer operations result in integer answers
		
		return i;
	}
	
	public static double findRem(int [] arr, int [] target) {
		double rem = 0;
		int max = 0;
		
		for(int i = 0; i < arr.length; i++) {
			if(arr[i] > max) {
				max = arr[i];
			}
		}
		//counting number of values held by attribute
		
		for(int i = 0; i <= max; i++) {
			int pi = 0, ni = 0;
			for(int j = 0; j < arr.length; j++) {
				if(arr[j] == i) {
					if(target[j] == 0) {
						pi++;
					}
					else {
						ni++;
					}
				}
			}
			//remainder = sum from i to v (((pi+ni)/(p+n)) * I((pi/(pi+ni)), (ni/(pi+ni)))) 
			rem += (((double)(pi+ni)/(double)(p+n)) * entropy(pi,ni));
		}
		
		return rem;
	}
	
	//converts string values of row to predict to corresponding int values from training data
	public static int[] turnInt(String [] arr, String [][] vals, int [][] numVal) {
		//arr is row to predict for
		//vals is training data string values
		//numVals is integer equivalent of training data
		
		int [] a = new int[arr.length];
		//a will store integer values of row to classify
		
		for(int i = 0; i < vals.length-1; i++) {
			//till length - 1 since one row of training data is the target
			for(int j = 0; j < vals[i].length-1; j++) {
				//till length - 1 since one column is the attribute name
				if(arr[i].equals(vals[i][j+1])) {
					//j+1 since the first column is the attribute name
					a[i] = numVal[i][j];
					break;
				}
				else {
					a[i] = -1;
					//if it is a new value then -1 (so we know its not present in training data)
				}
			}
		}
		
		return a;
	}

	//finds value of each attribute that occurs max times
	public static int[] conMax(int [][] arr) {
		//arr is the training data
		
		int [] array = new int[arr.length];
		//to store value that occurs most
		
		for(int i = 0; i < arr.length; i++) {
			//iterating through each row of training data
			
			int max = 0, maxVal = 0;
			//max stores the count of max value
			//maxVal stores actual value of the attribute
			
			for(int j = 0; j < arr[0].length; j++) {
				//iterating through each element of the current row
				
				int count  = 0;
				//stores count of each value
				
				for(int k = 0; k < arr[0].length; k++) {
					//iterating through elements of current row to compare to value at j
					
					if(arr[i][j] == arr[i][k]) {
						count++;
						if(count > max) {
							max = count;
							maxVal = arr[i][j];
						}
					}
				}
			}
			array[i] = maxVal;
		}
		
		return array;
	}
	
	
	//converting any value that isnt present in the training data to the value that occurs maximum times in that attribute
	public static int[] conv(int [] val, int [] max) {
		for(int i = 0; i < val.length; i++) {
			if(val[i] == -1) {
				//when converting to int values we made any value not in training data = -1
				val[i] = max[i];
			}
		}
		
		return val;
	}
	
	//returns classification of the row
	public static String predict(int [][] vals, int [] newVal, int pos, boolean [] done, int [] target, String tVals[], String s) {
		//vals is integer values of training data
		//pos is which row we are checking
		//done checks which values of each attribute to consider
		//target is integer array of target row
		//tVals is string values of target
		//String s is the value of target we are choosing
		
		int pi = 0, ni = 0;
		//holds number of positives and negatives for value of attribute
		
		int pPos = 0, nPos = 0;
		//stores position of last positive value and negative value respectively
		
		for(int i = 0; i < vals[0].length; i++) {
			//iterating through row of vals
			if(!done[i]) {
				//if we dont need to check this value
				continue;
			}
			
			if(vals[pos][i] == newVal[pos]) {
				//only checking values in training data that match the value we are trying to classify
				
				if(target[i] == 0) {
					//0 is positive
					pi++;
					pPos = i;
				}
				else {
					//1 is negative
					ni++;
					nPos = i;
				}
				done[i] = true;
				//in future iteration we will need to check the value at i
			}
			else {
				done[i] = false;
				//we should ignore the value at i in future iterations
			}
		}
		
		pos++;
		//increment pos to go to next row
		
		if(pi == 0 && ni == 0) {
			//we have reached the end of the tree
			return s;
		}
		else if(ni == 0) {
			//if no negative values of target then we classify as positive
			s = tVals[pPos+1];
			//pPos+1 since since first column is the row name
		}
		else if(pi == 0) {
			//if no positive values of target then we classify as negative
			s = tVals[nPos+1];
			//nPos+1 since since first column is the row name
		}
		else {
			//if both positive and negative values we need to check next attribute
			s = predict(vals, newVal, pos, done, target, tVals, s);
			//calling predict again with pos incremented earlier and done updated to know which values to ignore
		}
		return s;
	}
	
}