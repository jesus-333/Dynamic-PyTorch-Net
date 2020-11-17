function test(){
	var all_script = "";
	
	all_script = all_script + createIncipit();
	
	alert(all_script);
	
}


function createIncipit(){
	var incipit = "";
	
	incipit = incipit + "import torch\n";
	incipit = incipit + "from torch import nn\n";
	incipit = incipit + "import numpy as np\n";
	
	return incipit
}



