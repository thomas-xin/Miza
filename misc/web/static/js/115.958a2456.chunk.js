(this.webpackJsonpmizaweb2=this.webpackJsonpmizaweb2||[]).push([[115],{196:function(e,n){function a(e){return e?"string"===typeof e?e:e.source:null}function s(e){return i("(?=",e,")")}function i(){for(var e=arguments.length,n=new Array(e),s=0;s<e;s++)n[s]=arguments[s];var i=n.map((function(e){return a(e)})).join("");return i}e.exports=function(e){var n={className:"variable",variants:[{begin:/\$\d+/},{begin:/\$\{\w+\}/},{begin:i(/[$@]/,e.UNDERSCORE_IDENT_RE)}]},a={endsWithParent:!0,keywords:{$pattern:/[a-z_]{2,}|\/dev\/poll/,literal:["on","off","yes","no","true","false","none","blocked","debug","info","notice","warn","error","crit","select","break","last","permanent","redirect","kqueue","rtsig","epoll","poll","/dev/poll"]},relevance:0,illegal:"=>",contains:[e.HASH_COMMENT_MODE,{className:"string",contains:[e.BACKSLASH_ESCAPE,n],variants:[{begin:/"/,end:/"/},{begin:/'/,end:/'/}]},{begin:"([a-z]+):/",end:"\\s",endsWithParent:!0,excludeEnd:!0,contains:[n]},{className:"regexp",contains:[e.BACKSLASH_ESCAPE,n],variants:[{begin:"\\s\\^",end:"\\s|\\{|;",returnEnd:!0},{begin:"~\\*?\\s+",end:"\\s|\\{|;",returnEnd:!0},{begin:"\\*(\\.[a-z\\-]+)+"},{begin:"([a-z\\-]+\\.)+\\*"}]},{className:"number",begin:"\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}(:\\d{1,5})?\\b"},{className:"number",begin:"\\b\\d+[kKmMgGdshdwy]?\\b",relevance:0},n]};return{name:"Nginx config",aliases:["nginxconf"],contains:[e.HASH_COMMENT_MODE,{beginKeywords:"upstream location",end:/;|\{/,contains:a.contains,keywords:{section:"upstream location"}},{className:"section",begin:i(e.UNDERSCORE_IDENT_RE+s(/\s+\{/)),relevance:0},{begin:s(e.UNDERSCORE_IDENT_RE+"\\s"),end:";|\\{",contains:[{className:"attribute",begin:e.UNDERSCORE_IDENT_RE,starts:a}],relevance:0}],illegal:"[^\\s\\}\\{]"}}}}]);
//# sourceMappingURL=115.958a2456.chunk.js.map