# Laundering Money Prediction
This model tries to automate the manual decision process of a risk analyst to identify a possible case of money laundering from the events automatically raised by a software based on the behavior of the users of a platform. The model takes into account features such as the amount of dollars the user receives (*total_income_dollar_amount*), the amount of dollars the user sends (*total_outcome_dollar_amount*), and a risk scale automatically assigned to the user (risk_pld). From these numerical and categorical data, the current model must assign a probability (0-1) that the evaluated user is involved in money laundering.

Below I present a demo explaining the architecture of the solution and showing how it works.

https://www.loom.com/share/a99075eb512b45258db3181dec24b55d

## Architecture
-------------
![](https://i.ibb.co/m53pk9T/Model-store-and-model-registry.png)

## Features
-------------

- Support Standard Markdown / CommonMark and GFM(GitHub Flavored Markdown);
- Full-featured: Real-time Preview, Image (cross-domain) upload, Preformatted text/Code blocks/Tables insert, Code fold, Search replace, Read only, Themes, Multi-languages, L18n, HTML entities, Code syntax highlighting...;
- Markdown Extras : Support ToC (Table of Contents), Emoji, Task lists, @Links...;
- Compatible with all major browsers (IE8+), compatible Zepto.js and iPad;
- Support identification, interpretation, fliter of the HTML tags;
- Support TeX (LaTeX expressions, Based on KaTeX), Flowchart and Sequence Diagram of Markdown extended syntax;
- Support AMD/CMD (Require.js & Sea.js) Module Loader, and Custom/define editor plugins;


- [x] GFM task list 1
- [x] GFM task list 2
- [ ] GFM task list 3
    - [ ] GFM task list 3-1
    - [ ] GFM task list 3-2
    - [ ] GFM task list 3-3
- [ ] GFM task list 4
    - [ ] GFM task list 4-1
    - [ ] GFM task list 4-2

###Characters

**Emphasis**  __Emphasis__
***Emphasis Italic*** ___Emphasis Italic___




###Links

`EC2 Instance to check MLFLOW experiments` : [MLFlow](http:// http://ec2-18-206-253-206.compute-1.amazonaws.com:5000/#/experiments/3?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All Runs&selectedColumns=attributes.`Source`,attributes.`Models`,attributes.`Dataset`&compareRunCharts= "MLFlow") <http://ec2-18-206-253-206.compute-1.amazonaws.com:5000/#/experiments/3?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All Runs&selectedColumns=attributes.`Source`,attributes.`Models`,attributes.`Dataset`&compareRunCharts=>
`AWS EC2 Instance to make predictions` : 54.159.94.245/predict

###Code Blocks (multi-language) & highlighting

####Inline code

`$ npm install marked`

####Code Blocks (Indented style)

Body example:
```json
{
	"total_outcome_dollar_amount": [2109.19],
	"total_income_dollar_amount": [1305.06],
	"risk_pld": ["HIGH"]
}
```



###Lists

####Unordered list (-)

- Item A
- Item B
- Item C


####Ordered list

1. Item A
2. Item B
3. Item C

----

###Tables

First Header  | Second Header
------------- | -------------
Content Cell  | Content Cell
Content Cell  | Content Cell
