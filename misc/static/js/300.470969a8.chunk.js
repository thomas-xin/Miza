(this.webpackJsonpmizaweb2=this.webpackJsonpmizaweb2||[]).push([[300,112],{193:function(e,n){e.exports=function(e){var n={keyword:"if then not for in while do return else elseif break continue switch and or unless when class extends super local import export from using",literal:"true false nil",built_in:"_G _VERSION assert collectgarbage dofile error getfenv getmetatable ipairs load loadfile loadstring module next pairs pcall print rawequal rawget rawset require select setfenv setmetatable tonumber tostring type unpack xpcall coroutine debug io math os package string table"},s="[A-Za-z$_][0-9A-Za-z$_]*",i={className:"subst",begin:/#\{/,end:/\}/,keywords:n},a=[e.inherit(e.C_NUMBER_MODE,{starts:{end:"(\\s*/)?",relevance:0}}),{className:"string",variants:[{begin:/'/,end:/'/,contains:[e.BACKSLASH_ESCAPE]},{begin:/"/,end:/"/,contains:[e.BACKSLASH_ESCAPE,i]}]},{className:"built_in",begin:"@__"+e.IDENT_RE},{begin:"@"+e.IDENT_RE},{begin:e.IDENT_RE+"\\\\"+e.IDENT_RE}];i.contains=a;var t=e.inherit(e.TITLE_MODE,{begin:s}),r="(\\(.*\\)\\s*)?\\B[-=]>",o={className:"params",begin:"\\([^\\(]",returnBegin:!0,contains:[{begin:/\(/,end:/\)/,keywords:n,contains:["self"].concat(a)}]};return{name:"MoonScript",aliases:["moon"],keywords:n,illegal:/\/\*/,contains:a.concat([e.COMMENT("--","$"),{className:"function",begin:"^\\s*"+s+"\\s*=\\s*"+r,end:"[-=]>",returnBegin:!0,contains:[t,o]},{begin:/[\(,:=]\s*/,relevance:0,contains:[{className:"function",begin:r,end:"[-=]>",returnBegin:!0,contains:[o]}]},{className:"class",beginKeywords:"class",end:"$",illegal:/[:="\[\]]/,contains:[{beginKeywords:"extends",endsWithParent:!0,illegal:/[:="\[\]]/,contains:[t]},t]},{className:"name",begin:s+":",end:":",returnBegin:!0,returnEnd:!0,relevance:0}])}}},687:function(e,n,s){!function e(){e.warned||(e.warned=!0,console.log('Deprecation (warning): Using file extension in specifier is deprecated, use "highlight.js/lib/languages/moonscript" instead of "highlight.js/lib/languages/moonscript.js"'))}(),e.exports=s(193)}}]);
//# sourceMappingURL=300.470969a8.chunk.js.map