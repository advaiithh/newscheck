#!/usr/bin/perl
@codes = ("H","F","D","S","R","P","C");
open (FILE, "themes");
while ($line = <FILE> ) {
	chomp $line;
	($number,$themes) = split(/\t/,$line);
	$theme{$number}=$themes;
}

print qq+
\@relation fake

\@attribute text String
\@attribute H {1,0}
\@attribute F {1,0}
\@attribute D {1,0}
\@attribute S {1,0}
\@attribute R {1,0}
\@attribute P {1,0}
\@attribute C {1,0}
\@attribute class {Fake,Satire}


\@data

+;

open (LIST, "list2");
while ($file = <LIST>) {
	chomp $file;
	open (FILE, "$file");
	$fulltext="";
	while ($line = <FILE> ) {
		chomp $line;
		if (!($line =~ /^http/)) {
			$fulltext = $fulltext . " " . $line;
		}
	}
	close FILE;
	$fulltext =~ s/"//g;
	print "\"$fulltext\"";
	if ($file =~ /(\d+)\.txt/) {
		$number = $1;
		$themes = $theme{$number};
		foreach $code (@codes) {
			if ($themes =~ /$code/){ print "1"; } else {print "0";}
			print "\t";
		}
		
	}

	if ($file =~ /^Fake/) {
		print "Fake\n";
	}
	if ($file =~ /^Satire/) {
		print "Satire\n";
	}
		
}
close LIST;