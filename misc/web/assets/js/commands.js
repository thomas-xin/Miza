let commandsCache = {
};
async function grabCommands() {
	if (commandsCache && Object.keys(commandsCache).length) {
		return commandsCache;
	}
	let commands;
	try {
		let req = await fetch('/static/HELP.json');
		commandsCache = commands = await req.json();
	} catch(e) {
		commandsCache = {};
		return {};
	}
	return commands;
}

function shuffleArray(array) {
	for (let i = array.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[array[i], array[j]] = [array[j], array[i]];
	}
}

function withElement(selector, cb, parent=document) {
	let elem = parent.querySelector(selector);
	if (elem) {
		cb(elem);
		return elem;
	}
}

let prevBackListener = null;
async function prepareIndex() {
	const commands = await grabCommands();

	{
		// Url shenanigans
		let currentUrlParams = new URL(window.location).searchParams;
		let expectedUrl = new URL(window.location);
		expectedUrl.searchParams.delete('category');
		expectedUrl.searchParams.delete('cmd');
		if (currentUrlParams.get('cmd') || currentUrlParams.get('category')) {
			history.pushState({}, "", expectedUrl.toString());
		}
	}

	withElement('#go-back-link', goBackLink => {
		goBackLink.innerText = '';
		goBackLink.href = '';
		if (prevBackListener) {
			goBackLink.removeEventListener('click', prevBackListener);
		}
		prevBackListener = null;
	})

	withElement('.commands-index > h1', heading => {
		heading.innerText = 'Commands';
	});
	withElement('#command', commandHolder => {
		commandHolder.replaceChildren();
	})


	const categoriesHolder = document.getElementById('categories');
	categoriesHolder.replaceChildren(); // clears out loading message
	document.getElementById('command')?.replaceChildren?.();

	for (let catKey in commands) {
		let cmdElem = document.createElement('li');

		let cmdLink = document.createElement('a');

		let cmdHref = new URL(window.location);
		cmdHref.searchParams.set('category', catKey);
		cmdLink.classList.add('bg-href');
		cmdLink.href = cmdHref;
		cmdLink.addEventListener('click', e => {
			e.preventDefault();
			prepareCategory(catKey);
		})

		let cmdHead = document.createElement('span');
		cmdHead.classList.add('cat-header');
		// this can cause unicode bugs, but these should be ascii anyways
		cmdHead.innerText = catKey.toUpperCase()[0] + catKey.toLocaleLowerCase().substring(1);

		let cmdBody = document.createElement('span');
		cmdBody.classList.add('child-commands');

		for (let childCommand in commands[catKey]) {
			let childElem = document.createElement('a');
			cmdHref.searchParams.set('cmd', childCommand);
			childElem.href = cmdHref;
			childElem.innerText = childCommand;
			childElem.addEventListener('click', e => {
				e.preventDefault();
				prepareCommand(catKey, childCommand);
			})
			cmdBody.appendChild(childElem);
		}

		cmdElem.appendChild(cmdHead);
		cmdElem.appendChild(cmdBody);
		cmdElem.appendChild(cmdLink);
		categoriesHolder.appendChild(cmdElem);
	}
}

async function prepareCategory(catKey) {
	const commands = await grabCommands();
	const categoriesHolder = document.getElementById('categories');

	{
		// Url shenanigans
		let currentUrlParams = new URL(window.location).searchParams;
		let expectedUrl = new URL(window.location);
		expectedUrl.searchParams.set('category', catKey);
		expectedUrl.searchParams.delete('cmd');
		if (currentUrlParams.get('cmd') || currentUrlParams.get('category') !== catKey) {
			history.pushState({}, "", expectedUrl.toString());
		}
	}

	
	withElement('.commands-index > h1', heading => {
		heading.innerText = catKey.toUpperCase()[0] + catKey.toLocaleLowerCase().substring(1);
	});
	withElement('#command', commandHolder => {
		commandHolder.replaceChildren();
	})
	withElement('#go-back-link', goBackLink => {
		let goBackHref = new URL(window.location);
		goBackHref.searchParams.delete('category');
		goBackHref.searchParams.delete('cmd');
		goBackLink.innerText = "categories/";
		goBackLink.href = goBackHref;
		if (prevBackListener) {
			goBackLink.removeEventListener('click', prevBackListener);
		}
		prevBackListener = (e) => {
			e.preventDefault();
			prepareIndex();
		}
		goBackLink.addEventListener('click', prevBackListener);
	})

	categoriesHolder.replaceChildren(); // clears out loading message

	const category = commands[catKey];
	const cmdBodyTemplate = document.getElementById("command-short-description");

	for (let cmdKey in category) {
		let cmdElem = document.createElement('li');

		let cmdLink = document.createElement('a');

		let cmdHref = new URL(window.location);
		cmdHref.searchParams.set('category', catKey);
		cmdHref.searchParams.set('cmd', cmdKey);
		cmdLink.classList.add('bg-href');
		cmdLink.href = cmdHref;
		cmdLink.addEventListener('click', e => {
			e.preventDefault();
			prepareCommand(catKey, cmdKey);
		})
		
		let cmdHead = document.createElement('span');
		cmdHead.classList.add('cat-header');
		cmdHead.innerText = cmdKey;

		// Clone the new row and insert it into the table
		let cmdBody = document.importNode(cmdBodyTemplate.content, true);
		let command = category[cmdKey];
		let aliases = ('aliases' in command && command.aliases) ? command.aliases : [];

		aliases = aliases.filter(e => e !== cmdKey);

		let macros = ('macros' in command && command.macros) ? command.macros : {};
		aliases = aliases.filter(e => !Object.hasOwn(macros, e));

		if (aliases.length > 0) {
			let akaList = cmdBody.querySelector('.cmd-aka');
			if (akaList) {
				akaList.innerText = aliases.join(', ');
			}
		} else {
			let aka = cmdBody.querySelector('.aka');
			if (aka) {
				aka.parentElement.removeChild(aka);
			}
		}

		if ('description' in command && command.description) {
			let descElem = cmdBody.querySelector('.description');
			if (descElem) {
				descElem.innerText = command.description;
			}
		}

		cmdElem.appendChild(cmdHead);
		cmdElem.appendChild(cmdBody);
		cmdElem.appendChild(cmdLink);
		categoriesHolder.appendChild(cmdElem);
	}
}

function makeUpExampleForArgument(item) {
	switch (item.type) {
		case 'user':
			return '@someguy';
		case 'role':
			return '@somerole';
		case 'enum':
			return '--' + item.validation.enum[Math.floor(Math.random() * item.validation.enum.length)];
		case 'bool':
			return Math.random() < 0.5 ? 'true' : 'false';
		default:
			if (item?.example?.includes?.(' ') || item.greedy) {
				return ` "example text"`
			} else {
				return ` example_text`
			}
	}
}

async function prepareCommand(catKey, cmdKey) {
	const commands = await grabCommands();
	window.__commands = commands;
	{ 
		// Url shenanigans
		let currentUrlParams = new URL(window.location).searchParams;
		let expectedUrl = new URL(window.location);
		expectedUrl.searchParams.set('cmd', cmdKey);
		expectedUrl.searchParams.set('category', catKey);
		if (currentUrlParams.get('cmd') !== cmdKey || currentUrlParams.get('category') !== catKey) {
			history.pushState({}, "", expectedUrl.toString());
		}
	}

	withElement('#go-back-link', goBackLink => {
		let goBackHref = new URL(window.location);
		goBackHref.searchParams.delete('cmd');
		goBackLink.innerText = catKey.toUpperCase()[0] + catKey.toLocaleLowerCase().substring(1) + '/';
		goBackLink.href = goBackHref;
		if (prevBackListener) {
			goBackLink.removeEventListener('click', prevBackListener);
		}
		prevBackListener = (e) => {
			e.preventDefault();
			prepareCategory(catKey);
		}
		goBackLink.addEventListener('click', prevBackListener);
	})

	withElement('.commands-index > h1', heading => {
		heading.innerText = cmdKey;
	})

	withElement('#categories', categoriesHolder => categoriesHolder.replaceChildren());

	const commandHolder = document.getElementById('command');

	const category = commands[catKey];
	const cmdBodyTemplate = document.getElementById("command-body");

	// Clone the new row and insert it into the table
	let cmdBody = document.importNode(cmdBodyTemplate.content, true);
	let command = category[cmdKey];
	let aliases = ('aliases' in command && command.aliases) ? command.aliases : [];

	aliases = aliases.filter(e => e !== cmdKey);

	let macros = ('macros' in command && command.macros) ? command.macros : {};
	aliases = aliases.filter(e => !Object.hasOwn(macros, e));

	if (aliases.length > 0) {
		withElement('.cmd-aka', akaList => {
			akaList.replaceChildren(
				...aliases.flatMap((e, i) => {
					let elem = document.createElement('span');
					elem.classList.add('select');
					elem.innerText = e;
					if (i === 0) {
						return elem;
					}
					return [', ', elem];
				})
			);
		}, cmdBody);
	} else {
		withElement('.aka', aka => {
			aka.parentElement.removeChild(aka);
		}, cmdBody);
	}

	if ('description' in command && command.description) {
		withElement('.description', descElem => {
			descElem.innerText = command.description
		}, cmdBody);
	}

	if ('level' in command && typeof command.level === 'string') {
		withElement('.required-level', levelElem => {
			levelElem.innerText = command.level;
		}, cmdBody);
	}

	if ('rate_limit' in command && typeof command.rate_limit === 'string') {
		withElement('.rate-limit', rateElem => {
			let rate_limit = parseInt(command.rate_limit.split(',')[0].replace('(', ''), 10);

			let minutes = Math.floor(rate_limit / 60);
			let seconds = Math.floor(rate_limit % 60);
			if (minutes > 0) {
				rateElem.innerText = `${minutes}m `;
			} else {
				rateElem.innerText = '';
			}

			rateElem.innerText += `${seconds}s`;
		}, cmdBody);
	}

	if ('timeout' in command && typeof command.timeout === 'string') {
		withElement('.exec-timeout', timeoutElem => {
			let timeout = parseInt(command.timeout, 10);

			let minutes = Math.floor(timeout / 60);
			let seconds = Math.floor(timeout % 60);
			if (minutes > 0) {
				timeoutElem.innerText = `${minutes}m `;
			} else {
				timeoutElem.innerText = '';
			}

			timeoutElem.innerText += `${seconds}s`;
		}, cmdBody);
	}

	// Prepare schema
	let schema_unordered = ('schema' in command ? command.schema : null);
	let usage = ('usage' in command ? command.usage : null);
	let argument_ordering = ('ordered_args' in command ? command.ordered_args : null);
	let schema = schema_unordered;

	if (argument_ordering && schema) {
		schema = [];
		for (let schemakey of argument_ordering) {
			let schema_item = schema_unordered[schemakey];
			if (!schema_item) {
				continue;
			}

			schema_item.label = schemakey;
			schema.push(schema_item);
		}
	}

	if (schema) {
		withElement('.usage-head', headE => {
			headE.replaceWith();
		}, cmdBody);
		withElement('.arguments', argsE => {
			for (let argument of schema) {
				let argE = document.createElement('li');

				let labelE = document.createElement('label');
				labelE.innerText = argument.label;
				argE.appendChild(labelE);

				if (argument.required) {
					let requiredE = document.createElement('span');
					requiredE.classList.add('type');
					requiredE.innerText = 'required';
					argE.appendChild(requiredE);
				}

				let typeE = document.createElement('span');
				typeE.classList.add('type');
				typeE.innerText = argument.type;
				argE.appendChild(typeE);

				if (argument.description) {
					let descE = document.createElement('p');
					descE.innerText = argument.description;
					argE.appendChild(descE);
				}

				if (argument.validation) {
					let validLabelE = document.createElement('p');
					validLabelE.classList.add('validation-label');
					argE.appendChild(validLabelE);
					switch (argument.type) {
						case 'enum':
							if (!(argument.validation?.enum instanceof Array)) {
								break;
							}

							validLabelE.innerText = 'Must be one of:';

							let validsContainer = document.createElement('ul');
							validsContainer.classList.add('valid-enum-items');
							for (let item of argument.validation.enum) {
								let itemE = document.createElement('li');
								itemE.innerText = item;

								if (argument.default && item === argument.default) {
									itemE.classList.add('default');
									itemE.title = "Default";
								}

								validsContainer.appendChild(itemE);
							}
							argE.appendChild(validsContainer);

							break;
						default:
							// Assume it's a range thing
							let minmax = argument.validation.replace(/, ?/, ' and ');
							validLabelE.innerText = `Must be between ${minmax}.`;
							break;
					}
				}

				if (!argument.validation && argument.default) {
					let defaultsE = document.createElement('p');
					defaultsE.classList.add('validation-label');
					defaultsE.innerText = 'Defaults to ' + argument.default;
					argE.appendChild(defaultsE);
				}

				if (argument.excludes && argument.excludes instanceof Array) {
					let validLabelE = document.createElement('p');
					validLabelE.classList.add('validation-label');
					validLabelE.innerText = 'Cannot be used with ' + argument.excludes.join(', ') + '.';
					argE.appendChild(validLabelE);
				}

				argsE.appendChild(argE);
			}
		}, cmdBody);
	} else if (usage !== null) {
		withElement('.args-head', headE => {
			headE.replaceWith();
		}, cmdBody);
		withElement('.usage-str', elem => {
			elem.innerText = '~' + cmdKey + ' ' + usage;
		}, cmdBody);
	}

	if (command.example instanceof Array && command.example.length > 0) {
		withElement('.example-commands', elem => {
			elem.replaceChildren(
				...command.example.map(example => {
					let exampleElement = document.createElement('li')
					exampleElement.classList.add('select');
					exampleElement.innerText = '~' + example;

					return exampleElement;
				})
			)
		}, cmdBody);
	} else if (schema) {
		withElement('.example-commands', elem => {
			let exampleArgCount = 0;
			let excludesCount = 0;
			let exampleElements = [];

			for (let item of schema) {
				if (item.example) {
					exampleArgCount ++;
				}
				if (item.excludes && item.excludes instanceof Array) {
					excludesCount ++;
				}
			}
			if (exampleArgCount === 0 && schema.length > 0) {
				// make something up, I guess?

				let exampleCommand = '~' + cmdKey;
				let excludedArgs = new Map();
				let exclusionNum = 0;
				let usedSchema = schema;
				for (let item of usedSchema) {
					if (excludedArgs.has(item.label)) {
						continue;
					}

					if (item.excludes && item.excludes instanceof Array) {
						if (exclusionNum++ !== excludedItemNum) {
							continue;
						}
						item.excludes.forEach(e => excludedArgs.set(e, true));
					}

					exampleCommand += ' ';

					exampleCommand += makeUpExampleForArgument(item);
				}
				let exampleElement = document.createElement('li')
				exampleElement.classList.add('select');
				exampleElement.innerText = exampleCommand;

				elem.replaceChildren(exampleElement);
				return;
			}

			// Runs:
			// - Once for all exclusion permutations
			// - Once with expanded variables in reverse ordering
			for (let iterCount = 0; iterCount < 2; iterCount ++) {
				let expandVariables = iterCount === 1;
				for (let excludedItemNum = 0; excludedItemNum < excludesCount || excludesCount === 0; excludedItemNum++) {
					let exampleCommand = '~' + cmdKey;
					let excludedArgs = new Map();
					let exclusionNum = 0;
					let usedSchema = schema;
					if (expandVariables) {
						usedSchema = structuredClone(schema);
						usedSchema.reverse();
					}
					for (let item of usedSchema) {
						if (excludedArgs.has(item.label)) {
							continue;
						}
						if (item.example) {
							if (item.excludes && item.excludes instanceof Array) {
								if (exclusionNum++ !== excludedItemNum) {
									continue;
								}
								item.excludes.forEach(e => excludedArgs.set(e, true));
							}

							if (item.type === 'enum') {
								if (item.multiple) {
									exampleCommand += ' ' + item.example.split(' ').map(e => (`--${item.label} ${e}`)).join(' ');
								} else {
									exampleCommand += ` --${item.example}`
								}
								continue;
							}

							if (expandVariables) {
								exampleCommand += ` --${item.label}`
							}

							exampleCommand += ' ';
							switch (item.type) {
								case 'user':
									exampleCommand += '@someguy';
									break;
								case 'role':
									exampleCommand += '@somerole';
									break;
								default:
									if (item.example.includes(' ') || item.greedy) {
										exampleCommand += ` "${item.example}"`
									} else {
										exampleCommand += ` ${item.example}`
									}
							}
						} else if (item.required) {
							exampleCommand += ' ' + makeUpExampleForArgument(item);
						}
					}
					let exampleElement = document.createElement('li')
					exampleElement.classList.add('select');
					exampleElement.innerText = exampleCommand;

					if (expandVariables) {
						let noteElement = document.createElement('li');
						noteElement.classList.add('note');
						noteElement.innerText = 'Arguments are non-positional if explicitly named:';
						exampleElements.push(noteElement);
					}
					exampleElements.push(exampleElement);

					if (exampleArgCount < 2) {
						break;
					}
					if (excludesCount === 0) {
						break;
					}
					if (expandVariables) {
						break;
					}
				}

				if (exampleArgCount < 2) {
					break;
				}
			}

			elem.replaceChildren(...exampleElements);
		}, cmdBody);
	} else {
		withElement('.examples-head', headE => {
			headE.replaceWith();
		}, cmdBody);
	}

	if (macros && Object.keys(macros).length > 0) {
		withElement('.macros', argsE => {
			let shown = {};
			for (let key in macros) {
				let macroValue = macros[key];
				let stringlyMacroValue = JSON.stringify(macroValue);
				if (shown[stringlyMacroValue]) {
					continue;
				}
				shown[stringlyMacroValue] = true;

				// always guaranteed to be at least 1 length
				let similarMacros = Object.entries(macros).filter(e => JSON.stringify(e[1]) === stringlyMacroValue).map(e => e[0]);

				let macroE = document.createElement('li');
				let labelE = document.createElement('label');
				if (similarMacros.length === 1) {
					labelE.classList.add('select');
					labelE.innerText = '~' + similarMacros[0];
				} else {
					labelE.replaceChildren(
						`{ `,
						...similarMacros.flatMap((e, i) => {
							let elem = document.createElement('span');
							elem.classList.add('select');
							elem.innerText = '~' + e;
							if (i === 0) {
								return elem;
							}
							return [', ', elem];
						}),
						` }`
					);
				}
				macroE.appendChild(labelE);

				let descE = document.createElement('p');
				let selectableCmd = document.createElement('span');
				selectableCmd.classList.add('select');
				// This could be improved such that it's more accurate on quoting
				// but I don't want to write a parser for that.
				// So. Maybe later.
				selectableCmd.innerText = '~' + cmdKey + ' ' + Object.entries(macroValue).map(e => `--${e[0]} ${e[1]}`).join(' ');

				descE.replaceChildren(
					'↪ ', 
					selectableCmd
				);
				macroE.appendChild(descE);
				argsE.appendChild(macroE);
			}
		}, cmdBody);
	} else {
		withElement('.macros-head', headE => {
			headE.replaceWith();
		}, cmdBody);
	}

	commandHolder.replaceChildren(cmdBody);
}

async function runSetupFuncs() {
	let commands = await grabCommands();
	let url = new URL(window.location);
	let cat = url.searchParams.get('category');
	let cmd = url.searchParams.get('cmd');
	if (cat && Object.hasOwn(commands, cat)) {
		if (cmd && Object.hasOwn(commands[cat], cmd)) {
			await prepareCommand(cat, cmd);
		} else {
			await prepareCategory(cat);
		}
	} else {
		await prepareIndex();
	}
}

if (document.readyState === 'loading') {
	document.addEventListener('DOMContentLoaded', runSetupFuncs);
} else {
	runSetupFuncs();
}