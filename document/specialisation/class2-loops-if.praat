#writeInfoLine: "all spoken words"

#lastint = Get number of intervals: 5

# this is a forloop

#for interval from 1 to lastint
#	text$ = Get label of interval: 5, interval
#	appendInfoLine: text$
#endfor

# if and else

#writeInfoLine: "info about age"

#age = 30

#if age > 30
#	appendInfoLine: "this person is over 30"
#elsif age < 30
#	appendInfoLine: "this person is under 30"
#else ; if neither of the conditions are true
#	appendInfoLine: "this person is exacty 30"
#endif

# operators: =, <, >, !=, <=, >=, and, or, <> (not equal)


#forms

writeInfoLine: ""

form
	integer page
	integer: "Your age", "25" ; for default (having them as string can do Your age but variable your_age)
	positive: "height", "1.70"
	natural age2
	real age3
	boolean: "student", "no"
	choice: "subject", 1 ; can writing optionmenu instead of choice
		option: "speech"
		option: "sign"
endform

appendInfoLine: page
appendInfoLine: your_age
appendInfoLine: height
appendInfoLine: age2
appendInfoLine: age3
appendInfoLine: student

if subject = 1
	appendInfoLine: "speech"
else
	appendInfoLine: "sign"
endif



# a1

writeInfoLine: ""
num = 0
for num 1 to 10
    num = num + 1
    appendInfoLine: 2* num
endfor
